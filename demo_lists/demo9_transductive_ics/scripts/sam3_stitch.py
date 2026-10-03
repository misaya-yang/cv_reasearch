#!/usr/bin/env python3
"""Frozen SAM3 on paired episodes, preserving the FSS-SAM3 layout/inference with a
legal semantic-reference-mask-derived exemplar. A baseline reading, not a method.

  GPU, one shard:  python scripts/sam3_stitch.py --manifest SUITE/confirm_episodes.json --sam3 SRC --checkpoint sam3.pt \
                       --out results/sam3_v0/confirm --shard 0/2
  CPU after all predictions freeze: python scripts/sam3_stitch.py --score --manifest ... --out results/sam3_v0/confirm
  CPU table: python scripts/sam3_stitch.py --merge --manifest ... --out results/sam3_v0/confirm [--foris infer_1_confirm/episodes.jsonl]
  CPU, no weights: python scripts/sam3_stitch.py --fixture

Their 1-shot COCO setting, kept: 1008 x 1008 canvas, reference in the lower 60%, query in the upper 40% (each resized
to its rectangle without keeping the aspect ratio), the box of ONE connected component of the reference mask (area
above 1000 pixels, drawn at random; else the largest) as a positive exemplar, score threshold 0.5, union of the kept
masks, the query rectangle cut out and resized to the query's size with nearest neighbour, FP32.
Arms: `visual` (box only; public processor inserts dummy text "visual") and `text`
(true class name: privileged diagnostic, NEVER a method score). Current public
FSS-SAM3 b82ba838 evaluate.py imports data/dataset_tool.py, whose reference box is
a NONCROWD INSTANCE annotation of area>1000. We instead use the already-frozen
paired episodes and a legal semantic-mask component. This declared adaptation
does not exactly reproduce the published episode/instance-box protocol.
For the ledger each record also keeps the 20 most confident proposals of the `visual` arm before the threshold
(score, area and overlap with the truth inside the query rectangle, area and overlap with the reference mask inside
the reference rectangle) and the packed predicted mask.

Card, written 2026-10-03 before any SAM3 run of this project.
  Assumption: frozen SAM3 with the stitched canvas reproduces the published numbers on our image-isolated episodes.
  Prediction: on CONFIRM600 (FoRIS 59.78 there, 60.9 published, so the set is about 1 point harder) `visual` 63 to 67
    and `text` 72 to 77; on DEV241 (FoRIS 59.12) within the same ranges with twice the noise.
  Match: the baseline is reproduced; the ledger of its errors (nothing found / wrong object / part of the object /
    too much) is then read from the saved proposals, and nothing is proposed before that ledger exists.
  Mismatch: visual below60 holds confirmation for THREE public-evaluator checks;
    it does not automatically prove an implementation error. Above70 holds any
    claim pending the standard episode list. No extra stages are auto-created.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

CANVAS, RATIO, KEEP = 1008, 0.6, 20
ARMS = ("visual", "text")
FSS_COMMIT = "b82ba838678ae8c2f27f618d9500a50c171e223e"
CHECKPOINT_SHA256 = "9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e"
CHECKPOINT_BYTES = 3450062241
NAMES = ("person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,"
         "dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,"
         "baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,"
         "orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,"
         "cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush").split(",")
assert len(NAMES) == 80, len(NAMES)


def rectangles():
    """(left, top, width, height) of the reference and of the query on the canvas: COCO 1-shot layout of FSS-SAM3."""
    split = int(CANVAS * RATIO)
    return (0, CANVAS - split, CANVAS, split), (0, 0, CANVAS, CANVAS - split)


def component_box(mask, rng):
    """[x, y, w, h] of one 8-connected component of a binary mask: drawn among those above 1000 pixels, else the largest."""
    import numpy as np
    from scipy import ndimage
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return [0.0, 0.0, 0.0, 0.0]
    area = ndimage.sum(mask, lab, range(1, n + 1))
    big = [i for i in range(n) if area[i] > 1000]
    pick = rng.choice(big) if big else int(np.argmax(area))
    ys, xs = ndimage.find_objects(lab)[pick]
    return [float(xs.start), float(ys.start), float(xs.stop - xs.start), float(ys.stop - ys.start)]


def stitch(ref, query, box):
    """The canvas and the reference box on it as normalised [cx, cy, w, h]."""
    from PIL import Image
    canvas = Image.new("RGB", (CANVAS, CANVAS), (0, 0, 0))
    s, t = rectangles()
    canvas.paste(ref.resize(s[2:], Image.BILINEAR), s[:2])
    canvas.paste(query.resize(t[2:], Image.BILINEAR), t[:2])
    sx, sy = s[2] / ref.size[0], s[3] / ref.size[1]
    x, y, w, h = box
    return canvas, [(x * sx + s[0] + w * sx / 2) / CANVAS, (y * sy + s[1] + h * sy / 2) / CANVAS, w * sx / CANVAS, h * sy / CANVAS]


def to_original(canvas_mask, size):
    """The query rectangle of a canvas mask [CANVAS, CANVAS] (numpy, uint8) at the query's size (width, height)."""
    import numpy as np
    from PIL import Image
    x, y, w, h = rectangles()[1]
    return np.asarray(Image.fromarray(canvas_mask[y:y + h, x:x + w]).resize(size, Image.NEAREST)) > 0


def on_canvas(mask, rect):
    """A binary mask at an image's size placed on its canvas rectangle (nearest neighbour), as [h, w] bool."""
    import numpy as np
    from PIL import Image
    return np.asarray(Image.fromarray(mask.astype(np.uint8)).resize(rect[2:], Image.NEAREST)) > 0


class Runner:
    """The processor of the SAM3 source, with the proposals before the score threshold kept."""

    def __init__(self, processor):
        self.p = processor

    def __call__(self, canvas, box, text=None):
        import torch
        p = self.p
        state = p.set_image(canvas)
        p.reset_all_prompts(state)
        if text == "":
            raise ValueError("Empty text is not the official visual-only dummy prompt")
        if text is not None:
            state = p.set_text_prompt(text, state)
        seen = {}
        inner = p.model.forward_grounding

        def spy(**kw):
            seen["out"] = inner(**kw)
            return seen["out"]
        p.model.forward_grounding = spy
        try:
            state = p.add_geometric_prompt(state=state, box=box, label=True)
        finally:
            p.model.forward_grounding = inner
        masks = state["masks"]
        union = masks[:, 0].any(0) if masks.shape[0] else torch.zeros((CANVAS, CANVAS), dtype=torch.bool, device=masks.device)
        out = seen["out"]
        prob = (out["pred_logits"].sigmoid() * out["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)[0]
        top = prob.argsort(descending=True, stable=True)[:KEEP]
        raw = torch.nn.functional.interpolate(out["pred_masks"][0, top][:, None].float(), (CANVAS, CANVAS), mode="bilinear", align_corners=False)[:, 0] > 0
        return union, prob[top].float().cpu().tolist(), raw, int(masks.shape[0]), float(out["presence_logit_dec"].sigmoid().flatten()[0])


def infer_episode(run, ref, query, ref_mask, c, rng):
    """Both frozen arms. NO query annotation or truth argument is accepted."""
    import numpy as np
    import torch
    s, t = rectangles()
    original_box = component_box(ref_mask, rng)
    if original_box[2] <= 0 or original_box[3] <= 0:
        raise ValueError("Empty reference concept; no whole-image exemplar substitution")
    canvas, box = stitch(ref, query, original_box)
    rec = dict(pred_area={}, kept={}, presence={}, reference_area=float(ref_mask.mean()),
               query_shape=[query.height, query.width], exemplar_box=original_box,
               exemplar_box_normalized=box, visual_text="processor dummy visual, no category text",
               text_role="true class name; privileged diagnostic condition",
               components=int(__import__("scipy.ndimage").ndimage.label(ref_mask, structure=np.ones((3, 3)))[1]))
    packed = {}
    for arm in ARMS:
        union, prob, raw, kept, presence = run(canvas, box, NAMES[c] if arm == "text" else None)
        pred = to_original(union.cpu().numpy().astype(np.uint8), query.size)
        packed[arm] = np.packbits(pred)
        rec["pred_area"][arm], rec["kept"][arm], rec["presence"][arm] = int(pred.sum()), kept, presence
        if arm == "visual":
            ts = torch.from_numpy(on_canvas(ref_mask, s)).to(raw.device)
            q, r = raw[:, t[1]:t[1] + t[3], t[0]:t[0] + t[2]], raw[:, s[1]:s[1] + s[3], s[0]:s[0] + s[2]]
            rec["canvas_reference"] = int(ts.sum())
            rec["proposal_metadata"] = [[float(prob[i]), int(q[i].sum()), int(r[i].sum()), int((r[i] & ts).sum())] for i in range(len(prob))]
            reference_prediction = union[s[1]:s[1]+s[3], s[0]:s[0]+s[2]]
            rec["reference_union_iu"] = [int((reference_prediction & ts).sum()), int((reference_prediction | ts).sum())]
            # Only query-crop bitmaps are needed after freeze. Reference GT is legal.
            packed["proposal_query"] = np.packbits(q.cpu().numpy().reshape(len(prob), t[2]*t[3]), axis=1)
            rec["proposal_shape"] = [len(prob), t[3], t[2]]
            rec["proposal_scope"] = "top20 confidence proposals; may omit additional kept proposals"
    return rec, packed


def score_prediction(rec, packed, truth):
    """CPU scoring of frozen predictions and EXACT top20 candidate unions."""
    import numpy as np
    rec = dict(rec)
    shape = tuple(rec["query_shape"])
    if truth.shape != shape:
        raise ValueError("Query annotation shape differs from frozen RGB shape")
    rec["truth_area"] = int(truth.sum())
    rec["original_iu"] = {}
    for arm in ARMS:
        pred = np.unpackbits(packed[arm])[:truth.size].reshape(shape).astype(bool)
        rec["original_iu"][arm] = [int((pred & truth).sum()), int((pred | truth).sum())]
    ps = tuple(rec["proposal_shape"])
    raw = np.unpackbits(packed["proposal_query"], axis=1)[:, :ps[1]*ps[2]].reshape(ps).astype(bool)
    tq = on_canvas(truth, rectangles()[1])
    rec["canvas_truth"] = int(tq.sum())
    rec["proposals"] = [[p[0], p[1], int((raw[i] & tq).sum()), p[2], p[3]] for i, p in enumerate(rec["proposal_metadata"])]
    def union_iu(indices):
        union = raw[indices].any(0) if len(indices) else np.zeros(tq.shape, bool)
        return [int((union & tq).sum()), int((union | tq).sum())]
    rec["proposal_union_iu"] = {str(t): union_iu([i for i, p in enumerate(rec["proposals"]) if p[0] > t]) for t in (.1, .3, .5, .7)}
    rec["proposal_majority_truth_union_iu"] = union_iu([i for i, p in enumerate(rec["proposals"]) if p[1] > 0 and p[2] > .5*p[1]])
    return rec


def episode(run, ref, query, ref_mask, truth, c, rng):
    """Existing synthetic-fixture API only; production uses infer_episode."""
    rec, packed = infer_episode(run, ref, query, ref_mask, c, rng)
    return score_prediction(rec, packed, truth), packed["visual"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def selected_rows(a, man):
    if len(a.shard) != 2 or a.shard[1] <= 0 or not 0 <= a.shard[0] < a.shard[1]:
        raise ValueError("shard must be index/count, count>0 and index<count")
    rows = list(enumerate(man["episodes"][:a.limit]))[a.shard[0]::a.shard[1]]
    keys = [(r["fold"], r["e"], r["c"]) for _, r in rows]
    if len(keys) != len(set(keys)) or not rows:
        raise ValueError("Nonempty unique (fold,e,c) scope required")
    if any(not 0 <= r["c"] < 80 or r["c"] % 4 != r["fold"] or r["support"] == r["query"] for _, r in rows):
        raise ValueError("Invalid COCO fold/category or identical S/Q")
    return rows


def protocol():
    return dict(fss_source_commit=FSS_COMMIT, public_inference_layout_preserved=True,
                exact_published_episode_reproduction=False,
                reference_exemplar="one 8-connected semantic-mask component; >1000 pixels else largest",
                public_loader_difference="evaluate imports dataset_tool: noncrowd instance bbox, different sampled episodes",
                box_randomness="random.Random(global manifest index); paired adaptation",
                visual_prompt="no set_text_prompt; public processor inserts dummy visual",
                text_role="privileged true category diagnostic", score_threshold=.5,
                layout="1008 square; reference lower60%, query upper40%", precision="FP32/TF32off; no fallback",
                confirm_exposure="600 paired images already scored by earlier Claude readouts/formula; not fresh confirmation")


def reuse_predictions(a, man, rows):
    """Annotation-free engineering reuse; never reads old scored episode records."""
    if not a.reuse_predictions_from:
        return {}
    source = Path(a.reuse_predictions_from)
    freeze_path = source / ("prediction_freeze_shard%d.json" % a.shard[0])
    freeze = json.loads(freeze_path.read_text())
    predictions = source / ("predictions_shard%d.jsonl" % a.shard[0])
    if (freeze["state"] != "PREDICTIONS_FROZEN" or freeze["query_annotation_opened"] is not False or
        freeze["source_sha256"] != sha256(__file__) or freeze["manifest_sha256"] != sha256(a.manifest) or
        freeze["predictions_jsonl_sha256"] != sha256(predictions) or freeze["protocol"] != protocol()):
        raise ValueError("Reusable prefix code/config/manifest identity changed")
    old_report = json.loads((source / ("report_shard%d.json" % a.shard[0])).read_text())
    if old_report.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise ValueError("Reusable prefix checkpoint identity changed")
    asset_index = {item["path"]: item for item in freeze["prediction_files"]}
    reusable = {}
    by_key = {(r["fold"], r["e"], r["c"]): r for r in map(json.loads, predictions.read_text().splitlines())}
    for index, row in rows:
        key = (row["fold"], row["e"], row["c"])
        if key not in by_key:
            continue
        old = by_key[key]
        if (old["query"] != row["query"] or old["support"] != row["support"] or
            old.get("global_manifest_index") != index):
            raise ValueError("Reusable prefix S/Q or component-box random seed changed")
        for name in ("prediction_file", "candidate_file"):
            asset = asset_index[old[name]]
            file = source / asset["path"]
            if not file.exists() or file.stat().st_size != asset["bytes"] or sha256(file) != asset["sha256"]:
                raise ValueError("Reusable prefix needs unchanged packed outputs; score smoke with --keep-candidates")
        reusable[key] = (old, source)
    return reusable


def partial_predictions(a, rows, stream_path, report_path):
    """Verified same-code partial prefix. Never silently discard a corrupt tail."""
    if not stream_path.exists():
        return [], None
    if not a.resume:
        raise ValueError("Existing predictions preserved; use --resume for same-identity partial prefix")
    previous = json.loads(report_path.read_text())
    if (previous.get("state") not in ("RUNNING", "ERROR") or
        previous.get("source_sha256") != sha256(__file__) or
        previous.get("manifest_sha256") != sha256(a.manifest) or
        previous.get("checkpoint_sha256") != CHECKPOINT_SHA256 or previous.get("protocol") != protocol()):
        raise ValueError("Partial run identity/state changed; HOLD and preserve the old attempt")
    recs = [json.loads(line) for line in stream_path.read_text().splitlines() if line]
    expected = [(index,r["fold"],r["e"],r["c"],r["support"],r["query"]) for index,r in rows[:len(recs)]]
    observed = [(r["global_manifest_index"],r["fold"],r["e"],r["c"],r["support"],r["query"]) for r in recs]
    if len(recs)>len(rows) or observed != expected:
        raise ValueError("Partial predictions are not the exact manifest prefix")
    for rec in recs:
        if sorted(x["path"] for x in rec["prediction_assets"]) != sorted((rec["prediction_file"],rec["candidate_file"])):
            raise ValueError("Partial case lacks both frozen outputs")
        for asset in rec["prediction_assets"]:
            path = Path(a.out)/asset["path"]
            if path.stat().st_size != asset["bytes"] or sha256(path) != asset["sha256"]:
                raise ValueError("Partial case asset changed; no silent recomputation")
    snapshot = report_path.with_name(report_path.stem+".before_resume_"+str(time.time_ns())+".json")
    snapshot.write_text(json.dumps(previous,indent=1))
    return recs, previous


def work(a):
    import numpy as np
    import torch
    from PIL import Image
    if not a.unguarded and os.environ.get("DEMO9_CUDA_GUARD") != "1":
        raise SystemExit("run under scripts/experiment_resource_guard.py, or pass --unguarded")
    man = json.loads(Path(a.manifest).read_text())
    out = Path(a.out)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    (out / "candidates").mkdir(parents=True, exist_ok=True)
    stream_path = out / ("predictions_shard%d.jsonl" % a.shard[0])
    rows = selected_rows(a, man)
    report_path = out / ("report_shard%d.json" % a.shard[0])
    completed, previous = partial_predictions(a, rows, stream_path, report_path)
    reusable = reuse_predictions(a, man, rows)
    rect = rectangles()[1]
    candidate_worst = len(rows)*KEEP*((rect[2]*rect[3]+7)//8)
    if candidate_worst > a.candidate_budget_bytes:
        raise ValueError("Finite candidate bitmap budget exceeded before model construction")
    if not torch.cuda.is_available():
        raise RuntimeError("Real CUDA device required; no CPU fallback")
    status = json.loads(Path(a.checkpoint_status).read_text())
    if (status.get("state") != "WEIGHT_VERIFIED" or status.get("actual_sha256") != CHECKPOINT_SHA256 or
        status.get("actual_bytes") != CHECKPOINT_BYTES or a.checkpoint.stat().st_size != CHECKPOINT_BYTES or
        Path(status["path"]).resolve() != a.checkpoint.resolve()):
        raise ValueError("Checkpoint differs from the verified official asset receipt")
    report = dict(state="RUNNING", episodes=len(completed), expected=len(rows), checkpoint=str(a.checkpoint),
                  checkpoint_sha256=CHECKPOINT_SHA256, manifest_sha256=sha256(a.manifest),
                  source_sha256=sha256(__file__), canvas=CANVAS, ratio=RATIO, precision="fp32",
                  memory_fraction=a.memory_fraction, query_annotation_opened=False, protocol=protocol(),
                  candidate_worst_case_packed_bytes=candidate_worst, candidate_budget_bytes=a.candidate_budget_bytes,
                  candidate_geometry="query crop only; legal reference intersections computed before GT",
                  reused_prefix_episodes=len(reusable), reused_from=str(a.reuse_predictions_from) if reusable else None)
    report["resumed_complete_cases"] = len(completed)
    start = time.monotonic()

    def save():
        report["elapsed_s"] = time.monotonic()-start+(previous.get("elapsed_s",0.) if previous else 0.)
        (out / ("report_shard%d.json" % a.shard[0])).write_text(json.dumps(report, indent=1))
    save()
    try:
        sys.path.insert(0, str(a.sam3))
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        torch.cuda.set_per_process_memory_fraction(a.memory_fraction)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(0)
        model_start = time.monotonic()
        model = build_sam3_image_model(bpe_path=str(Path(a.sam3) / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"), device="cuda",
                                       checkpoint_path=str(a.checkpoint), load_from_HF=False,
                                       eval_mode=True, enable_inst_interactivity=False, compile=False).float().eval()
        if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
            raise RuntimeError("FP32 model contract failed")
        run = Runner(Sam3Processor(model, device="cuda", confidence_threshold=.5))
        report["model_load_s"] = time.monotonic()-model_start
        data, ann = Path(man["data_root"]), Path(man["annotation_root"])
        prediction_files = [asset for rec in completed for asset in rec["prediction_assets"]]
        inference_start = time.monotonic()
        with open(stream_path, "a") as stream, torch.inference_mode():
            for i, row in rows[len(completed):]:
                c = row["c"]
                begin = time.monotonic()
                key = (row["fold"], row["e"], c)
                if key in reusable:
                    old, old_out = reusable[key]
                    rec = dict(old)
                    with np.load(old_out / old["prediction_file"], allow_pickle=False) as masks, np.load(old_out / old["candidate_file"], allow_pickle=False) as candidates:
                        packed = {arm: masks[arm] for arm in ARMS}
                        packed["proposal_query"] = candidates["proposal_query"]
                    rec.update(reused_prediction=True, reused_freeze_sha256=sha256(old_out / ("prediction_freeze_shard%d.json" % a.shard[0])))
                else:
                    ref, query = (Image.open(data / row[k]).convert("RGB") for k in ("support", "query"))
                    # ONLY the legal reference annotation opens in this GPU process.
                    ref_mask = np.asarray(Image.open(ann / Path(row["support"]).with_suffix(".png"))) == c+1
                    rec, packed = infer_episode(run, ref, query, ref_mask, c, random.Random(i))
                    rec.update(reused_prediction=False)
                rec.update(fold=row["fold"], e=row["e"], c=c, query=row["query"], support=row["support"], global_manifest_index=i, seconds=time.monotonic() - begin)
                stem = "%d_%d_%d.npz" % (row["fold"], row["e"], c)
                mask_path, candidate_path = out / "masks" / stem, out / "candidates" / stem
                if mask_path.exists() or candidate_path.exists():
                    raise ValueError("Preserve old assets: fresh stage output required")
                np.savez_compressed(mask_path, visual=packed["visual"], text=packed["text"], shape=np.array(rec["query_shape"]))
                np.savez_compressed(candidate_path, proposal_query=packed["proposal_query"])
                rec["prediction_file"], rec["candidate_file"] = str(mask_path.relative_to(out)), str(candidate_path.relative_to(out))
                rec["prediction_assets"] = [dict(path=str(path.relative_to(out)),bytes=path.stat().st_size,sha256=sha256(path)) for path in (mask_path,candidate_path)]
                prediction_files.extend(rec["prediction_assets"])
                stream.write(json.dumps(rec) + "\n")
                stream.flush()
                report["episodes"] += 1
                report["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
                save()
                print("%d/%d inference; %.2fs this episode" % (report["episodes"], len(rows), rec["seconds"]), flush=True)
        report["inference_s"] = time.monotonic()-inference_start
        new_count = sum((r["fold"],r["e"],r["c"]) not in reusable for _,r in rows[len(completed):])
        report["mean_episode_s"] = report["inference_s"]/max(new_count,1)
        report["new_inferred_cases"] = new_count
        receipt = dict(state="PREDICTIONS_FROZEN", shard=list(a.shard), episodes=len(rows),
                       manifest_sha256=report["manifest_sha256"], source_sha256=report["source_sha256"],
                       query_annotation_opened=False, arms=list(ARMS), prediction_files=prediction_files,
                       predictions_jsonl_sha256=sha256(stream_path), protocol=report["protocol"])
        freeze_path = out / ("prediction_freeze_shard%d.json" % a.shard[0])
        freeze_path.write_text(json.dumps(receipt, indent=1))
        report.update(state="PREDICTIONS_FROZEN", prediction_freeze_sha256=sha256(freeze_path))
    except BaseException as err:
        report.update(state="ERROR", error=repr(err))
        raise
    finally:
        save()


def score(a):
    """CPU stage: validate the WHOLE stage freeze before first query annotation."""
    import numpy as np
    from PIL import Image
    out = Path(a.out)
    man = json.loads(Path(a.manifest).read_text())
    rows = selected_rows(a, man)
    freeze_path = out / ("prediction_freeze_shard%d.json" % a.shard[0])
    report_path = out / ("report_shard%d.json" % a.shard[0])
    rec_path = out / ("predictions_shard%d.jsonl" % a.shard[0])
    stream_path = out / ("episodes_shard%d.jsonl" % a.shard[0])
    report, freeze = json.loads(report_path.read_text()), json.loads(freeze_path.read_text())
    if stream_path.exists() or report["state"] != "PREDICTIONS_FROZEN":
        raise ValueError("Fresh scoring output and PREDICTIONS_FROZEN required")
    if (freeze["state"] != "PREDICTIONS_FROZEN" or freeze["query_annotation_opened"] is not False or
        freeze["manifest_sha256"] != sha256(a.manifest) or freeze["predictions_jsonl_sha256"] != sha256(rec_path) or
        report["prediction_freeze_sha256"] != sha256(freeze_path)):
        raise ValueError("Prediction identity or query-GT embargo failed")
    recs = [json.loads(line) for line in rec_path.read_text().splitlines() if line]
    expected = [(r["fold"], r["e"], r["c"], r["support"], r["query"]) for _, r in rows]
    observed = [(r["fold"], r["e"], r["c"], r["support"], r["query"]) for r in recs]
    if observed != expected or freeze["episodes"] != len(rows):
        raise ValueError("Frozen episode scope differs from manifest")
    expected_assets = [r[k] for r in recs for k in ("prediction_file", "candidate_file")]
    observed_assets = [r["path"] for r in freeze["prediction_files"]]
    if len(observed_assets) != len(set(observed_assets)) or sorted(observed_assets) != sorted(expected_assets):
        raise ValueError("Freeze does not cover every arm/candidate file exactly once")
    for asset in freeze["prediction_files"]:
        path = out / asset["path"]
        if path.stat().st_size != asset["bytes"] or sha256(path) != asset["sha256"]:
            raise ValueError("Frozen prediction asset changed: "+str(path))
    # All preceding validation is annotation-free. Query annotations open ONLY NOW.
    start = time.monotonic()
    report.update(state="SCORING", query_annotation_opened=True, all_predictions_frozen_before_first_query_annotation=True)
    report_path.write_text(json.dumps(report, indent=1))
    try:
        with open(stream_path, "x") as stream:
            for rec in recs:
                truth = np.asarray(Image.open(Path(man["annotation_root"]) / Path(rec["query"]).with_suffix(".png"))) == rec["c"]+1
                with np.load(out / rec["prediction_file"], allow_pickle=False) as masks, np.load(out / rec["candidate_file"], allow_pickle=False) as candidates:
                    packed = {arm: masks[arm] for arm in ARMS}
                    packed["proposal_query"] = candidates["proposal_query"]
                    measured = score_prediction(rec, packed, truth)
                stream.write(json.dumps(measured)+"\n"); stream.flush()
        removal = []
        for asset in freeze["prediction_files"]:
            if asset["path"].startswith("candidates/") and not a.keep_candidates:
                (out / asset["path"]).unlink()
                removal.append(asset)
        cleanup = dict(state="SCORING_TEMPORARIES_RETAINED_FOR_REUSE" if a.keep_candidates else "SCORING_TEMPORARIES_REMOVED", removed=removal,
                       released_bytes=sum(x["bytes"] for x in removal),
                       freeze_receipt_sha256=sha256(freeze_path), scalar_ledger_sha256=sha256(stream_path),
                       masks_preserved=True, arbitrary_new_candidate_rules_not_reconstructible=True)
        (out / ("candidate_cleanup_shard%d.json" % a.shard[0])).write_text(json.dumps(cleanup, indent=1))
        report.update(state="COMPLETED", scored_episodes=len(recs), score_elapsed_s=time.monotonic()-start,
                      candidate_released_bytes=cleanup["released_bytes"], ledger_exact_unions=True)
    except BaseException as err:
        report.update(state="ERROR_SCORING", error=repr(err))
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=1))
    print(json.dumps(dict(state=report["state"], episodes=len(recs), freeze_before_query_GT=True, candidate_released_bytes=report["candidate_released_bytes"])))


def class_miou(recs, arm):
    acc = {}
    for r in recs:
        v = acc.setdefault(r["c"], [0, 0])
        v[0] += r["original_iu"][arm][0]
        v[1] += r["original_iu"][arm][1]
    return 100 * sum(i / max(u, 1) for i, u in acc.values()) / max(len(acc), 1)


def paired(recs, get, base, draws=2000):
    """Paired class-mIoU; shared S/Q photos resampled as connected groups, seed0."""
    import numpy as np
    cls = np.array([r["c"] for r in recs])
    a, b = np.array([get(r) for r in recs], float), np.array([base(r) for r in recs], float)
    ids = np.unique(cls)

    def score(v, pick):
        i, u, k = v[pick, 0], v[pick, 1], cls[pick]
        return 100 * np.mean([i[k == c].sum() / max(u[k == c].sum(), 1) for c in ids if (k == c).any()])
    if not recs:
        raise ValueError("No paired observations")
    parent = list(range(len(recs)))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    photo_owner = {}
    for index, record in enumerate(recs):
        for name in ("support", "query"):
            photo = record.get(name)
            if photo is None:  # synthetic fixtures have no photos
                continue
            if photo in photo_owner:
                parent[find(index)] = find(photo_owner[photo])
            else:
                photo_owner[photo] = index
    groups = {}
    for index in range(len(recs)):
        groups.setdefault(find(index), []).append(index)
    clusters = [np.asarray(g, int) for g in groups.values()]
    every, rng = np.arange(len(recs)), np.random.default_rng(0)
    boot = []
    for _ in range(draws):
        pick = np.concatenate([clusters[j] for j in rng.integers(0, len(clusters), len(clusters))])
        boot.append(score(a, pick)-score(b, pick))
    return dict(miou=score(a, every), gain=score(a, every)-score(b, every),
                ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if len(clusters)>1 else None,
                bootstrap_unit="connected support/query photograph groups", groups=len(clusters),
                largest_group=max(map(len, clusters)), draws=draws)


def paired_host(man, rows, foris):
    """Strict full-scope join; headers only, no annotation pixels opened."""
    from PIL import Image
    host = {}
    for record in map(json.loads, Path(foris).read_text().splitlines()):
        if "original_iu" not in record or "native" not in record["original_iu"]:
            raise ValueError("FoRIS record missing original-resolution native I/U")
        key = (record["fold"], record["e"], record["c"])
        if key in host:
            raise ValueError("Duplicate FoRIS episode key")
        host[key] = record
    dimensions = {}
    for row in rows:
        key = (row["fold"], row["e"], row["c"])
        if key not in host or any(host[key].get(k) != row[k] for k in ("support", "query")):
            raise ValueError("FoRIS missing episode or S/Q UID mismatch: "+str(key))
        for image_role in ("support", "query"):
            relative = row[image_role]
            with Image.open(Path(man["data_root"])/relative) as image:
                rgb_size = image.size
            with Image.open(Path(man["annotation_root"])/Path(relative).with_suffix(".png")) as image:
                annotation_size = image.size  # metadata only: no pixel load
            if rgb_size != annotation_size:
                raise ValueError("RGB/annotation original size differs: "+relative)
        dimensions[key] = list(reversed(rgb_size))
        iu = host[key]["original_iu"]["native"]
        if not 0 <= iu[0] <= iu[1] <= rgb_size[0]*rgb_size[1]:
            raise ValueError("FoRIS original I/U outside actual query dimensions")
    return host, dimensions


def preflight(a):
    man = json.loads(Path(a.manifest).read_text())
    rows = [row for _, row in selected_rows(a, man)]
    if not a.foris:
        raise ValueError("SAM3 preflight requires the complete paired FoRIS records")
    host, dimensions = paired_host(man, rows, a.foris)
    value = dict(state="CPU_PROTOCOL_CHECKED", episodes=len(rows), foris_complete_uid_coverage=True,
                 foris_duplicate_keys=False, RGB_annotation_original_dimensions_exact=True,
                 foris_original_iu_within_original_dimensions=True, query_annotation_pixels_opened=False,
                 manifest_sha256=sha256(a.manifest), foris_sha256=sha256(a.foris), source_sha256=sha256(__file__),
                 protocol=protocol(), gpu_or_full_model_constructed=False,
                 foris_saved_per_case_dimensions_present=all("query_shape" in host[(r["fold"],r["e"],r["c"])] for r in rows))
    if a.out:
        path = Path(a.out); path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise ValueError("Preserve old preflight receipt; fresh path required")
        path.write_text(json.dumps(value, indent=1))
    print(json.dumps(value))


def merge(a):
    import numpy as np
    out = Path(a.out)
    man = json.loads(Path(a.manifest).read_text())
    want = len(man["episodes"][:a.limit])
    recs = [json.loads(l) for f in sorted(out.glob("episodes_shard*.jsonl")) for l in open(f) if l.strip()]
    shards = [json.loads(f.read_text()) for f in sorted(out.glob("report_shard*.json"))]
    recs.sort(key=lambda r: (r["fold"], r["e"]))
    expected_rows = man["episodes"][:a.limit]
    expected = {(r["fold"],r["e"],r["c"]):(r["support"],r["query"]) for r in expected_rows}
    observed = [(r["fold"],r["e"],r["c"]) for r in recs]
    ok = (len(recs) == want and len(observed) == len(set(observed)) and set(observed) == set(expected)
          and shards and all(s["state"] == "COMPLETED" for s in shards)
          and all((r["support"],r["query"]) == expected[(r["fold"],r["e"],r["c"])] for r in recs))
    if not ok:
        partial = dict(state="PARTIAL", episodes=len(recs), expected=want, scientific_tables_withheld=True,
                       reason="requires completed shards, unique keys and exact full manifest S/Q UID coverage")
        (out / "report.json").write_text(json.dumps(partial, indent=1))
        raise ValueError(partial["reason"])
    rep = dict(state="COMPLETED", episodes=len(recs), expected=want, protocol=protocol(),
               scope="original-resolution class mIoU on frozen paired episodes; legal component-box SAM3 adaptation",
               confirmation_exposure="previously scored paired images, not independent fresh confirmation",
               class_miou={k: class_miou(recs, k) for k in ARMS}, per_fold={k: [class_miou([r for r in recs if r["fold"] == f], k) for f in range(4)] for k in ARMS})
    if a.foris:
        host, dimensions = paired_host(man, expected_rows, a.foris)
        both = recs
        if any(r["query_shape"] != dimensions[(r["fold"],r["e"],r["c"])] for r in recs):
            raise ValueError("SAM3 original prediction shape differs from exact paired image/annotation dimensions")
        key = lambda r: host[(r["fold"], r["e"], r["c"])]["original_iu"]["native"]
        rep["against_foris"] = dict(episodes=len(both), foris=paired(both, key, key)["miou"],
                                    exact_full_episode_and_SQ_coverage=True, same_original_image_dimensions=True,
                                    **{k: paired(both, lambda r, k=k: r["original_iu"][k], key) for k in ARMS})
        better = lambda r: max(r["original_iu"]["visual"], key(r), key=lambda v: v[0] / max(v[1], 1))
        rep["against_foris"]["better_of_visual_and_foris_per_episode_LABELS"] = paired(both, better, key)  # a selection gap, not headroom
        fi = np.array([key(r)[0] / max(key(r)[1], 1) for r in both])
        si = np.array([r["original_iu"]["visual"][0] / max(r["original_iu"]["visual"][1], 1) for r in both])
        rep["against_foris"]["episodes_by_outcome"] = dict(both_above_half=int(((fi > .5) & (si > .5)).sum()), only_foris=int(((fi > .5) & (si <= .5)).sum()),
                                                           only_sam3=int(((fi <= .5) & (si > .5)).sum()), neither=int(((fi <= .5) & (si <= .5)).sum()),
                                                           correlation=float(np.corrcoef(fi, si)[0, 1]) if len(both) > 2 else 0.0)
    # the ledger of the visual arm: where the lost intersection-over-union goes
    iou = lambda r: r["original_iu"]["visual"][0] / max(r["original_iu"]["visual"][1], 1)
    kinds = dict(nothing_kept=lambda r: r["pred_area"]["visual"] == 0, kept_but_no_overlap=lambda r: r["pred_area"]["visual"] > 0 and r["original_iu"]["visual"][0] == 0,
                 too_much=lambda r: iou(r) > 0 and r["pred_area"]["visual"] > 1.5 * r["truth_area"], too_little=lambda r: iou(r) > 0 and r["pred_area"]["visual"] < r["truth_area"] / 1.5)
    rep["ledger"] = {k: dict(episodes=int(sum(map(f, recs))), mean_iou=float(np.mean([iou(r) for r in recs if f(r)] or [0]) * 100)) for k, f in kinds.items()}
    rep["ledger"]["canvas_kept_but_query_empty"] = int(sum(r["kept"]["visual"]>0 and r["pred_area"]["visual"]==0 for r in recs))
    rep["ledger"]["mean_episode_iou"] = float(np.mean([iou(r) for r in recs]) * 100) if recs else 0.0
    # Actual kept-mask UNION, not a sum of overlapping/truncated proposal areas.
    def reference_iou(r):
        i, u = r["reference_union_iu"]
        return i/max(u, 1)
    if recs:
        ri, qi = np.array([reference_iou(r) for r in recs]), np.array([iou(r) for r in recs])
        rep["ledger"]["reference_reproduction"] = dict(mean_iou=float(ri.mean() * 100), below_half=int((ri < .5).sum()),
                                                       query_iou_when_below_half=float(qi[ri < .5].mean() * 100) if (ri < .5).any() else None,
                                                       query_iou_when_above_half=float(qi[ri >= .5].mean() * 100) if (ri >= .5).any() else None,
                                                       correlation_with_query_iou=float(np.corrcoef(ri, qi)[0, 1]) if len(recs) > 2 else 0.0)
        share = np.array([r["canvas_truth"] / (CANVAS * (CANVAS - int(CANVAS * RATIO))) for r in recs])
        edges = [0, .01, .03, .1, .3, 1.01]
        rep["ledger"]["by_target_share_of_query"] = {"%g-%g" % (lo, hi): dict(episodes=int(((share >= lo) & (share < hi)).sum()),
                                                     mean_iou=float(qi[(share >= lo) & (share < hi)].mean() * 100) if ((share >= lo) & (share < hi)).any() else None)
                                                     for lo, hi in zip(edges[:-1], edges[1:])}

    if recs:
        base = lambda r: r["proposal_union_iu"]["0.5"]
        rep["proposal_reading"] = dict(note="EXACT union of top20 proposals in query canvas crop; truncated family, different resolution from main table; privileged selection is not perfect-input oracle",
                                       threshold_0p5=paired(recs, base, base)["miou"],
                                       **{"threshold_%s" % t: paired(recs, lambda r, t=t: r["proposal_union_iu"][str(t)], base) for t in (.1, .3, .7)},
                                       proposals_mostly_on_target_LABELS=paired(recs, lambda r: r["proposal_majority_truth_union_iu"], base))
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    sys.exit(0 if ok else 1)


def fixture():
    """CPU checks of everything that is not the model: layout, box mapping, the way back to the query, the metric."""
    import numpy as np
    import torch
    from PIL import Image
    rng, checks = np.random.default_rng(0), []
    s, t = rectangles()
    checks.append(("rectangles tile the canvas", s[1] + s[3] == CANVAS and t[1] + t[3] == s[1] and s[2] == t[2] == CANVAS))
    for case, (rs, qs) in enumerate((((640, 480), (500, 375)), ((333, 500), (640, 427)), ((640, 640), (427, 640)))):
        ref = Image.fromarray(rng.integers(0, 255, (rs[1], rs[0], 3), dtype=np.uint8))
        query = Image.fromarray(rng.integers(0, 255, (qs[1], qs[0], 3), dtype=np.uint8))
        ref_mask = np.zeros((rs[1], rs[0]), bool)
        ref_mask[rs[1] // 4:rs[1] // 2, rs[0] // 5:rs[0] // 2] = True
        ref_mask[5:9, 5:9] = True  # a small second component that must not be chosen
        truth = np.zeros((qs[1], qs[0]), bool)
        truth[qs[1] // 3:qs[1] // 3 * 2, qs[0] // 4:qs[0] // 4 * 3] = True
        box = component_box(ref_mask, random.Random(0))
        checks.append(("case %d: the large component is the box" % case, box == [rs[0] // 5, rs[1] // 4, rs[0] // 2 - rs[0] // 5, rs[1] // 2 - rs[1] // 4]))
        canvas, norm = stitch(ref, query, box)
        placed = np.zeros((CANVAS, CANVAS), bool)
        big = ref_mask.copy()
        big[5:9, 5:9] = False
        placed[s[1]:s[1] + s[3], s[0]:s[0] + s[2]] = on_canvas(big, s)
        ys, xs = np.where(placed)
        want = [(xs.min() + xs.max() + 1) / 2 / CANVAS, (ys.min() + ys.max() + 1) / 2 / CANVAS, (xs.max() + 1 - xs.min()) / CANVAS, (ys.max() + 1 - ys.min()) / CANVAS]
        checks.append(("case %d: the box lands on the pasted reference mask within 2 pixels" % case, max(abs(u - v) for u, v in zip(norm, want)) * CANVAS <= 2))
        checks.append(("case %d: the query is pasted where the rectangle says" % case,
                       np.array_equal(np.asarray(canvas)[t[1]:t[1] + t[3]], np.asarray(query.resize(t[2:], Image.BILINEAR)))))

        class Fake:  # a model that answers with the truth on the canvas, and with one wrong proposal below the threshold
            def __call__(self, canvas, box, text=None):
                full = np.zeros((CANVAS, CANVAS), bool)
                full[t[1]:t[1] + t[3], t[0]:t[0] + t[2]] = on_canvas(truth, t)
                wrong = np.zeros_like(full)
                wrong[s[1]:s[1] + 50, :50] = True
                return torch.from_numpy(full), [0.9, 0.2], torch.from_numpy(np.stack([full, wrong])), 1, 0.95
        rec, packed = episode(Fake(), ref, query, ref_mask, truth, 3, random.Random(0))
        i, u = rec["original_iu"]["visual"]
        checks.append(("case %d: a perfect canvas answer comes back above 0.97 IoU" % case, i / u > 0.97))
        checks.append(("case %d: the saved mask unpacks to the scored mask" % case, int(np.unpackbits(packed)[:truth.size].sum()) == rec["pred_area"]["visual"]))
        checks.append(("case %d: the proposal rows are filled" % case, rec["proposals"][0][2] == rec["canvas_truth"] and rec["proposals"][1][2] == 0))
    recs = [dict(c=0, original_iu=dict(visual=[5, 10], text=[10, 10])), dict(c=0, original_iu=dict(visual=[5, 10], text=[10, 10])),
            dict(c=1, original_iu=dict(visual=[0, 10], text=[10, 10]))]
    checks.append(("class mIoU pools pixels inside a class", abs(class_miou(recs, "visual") - 25.0) < 1e-9))
    v = paired(recs, lambda r: r["original_iu"]["text"], lambda r: r["original_iu"]["visual"], draws=50)
    checks.append(("paired difference", abs(v["gain"] - 75.0) < 1e-9))
    checks.append(("80 class names, the 41st is wine glass", NAMES[40] == "wine glass" and NAMES[0] == "person" and NAMES[79] == "toothbrush"))
    for name, ok in checks:
        print("%s  %s" % ("ok  " if ok else "FAIL", name))
    print("%d/%d checks pass" % (sum(ok for _, ok in checks), len(checks)))
    sys.exit(0 if all(ok for _, ok in checks) else 1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest")
    p.add_argument("--out")
    p.add_argument("--sam3", type=Path, help="folder of the SAM3 source (contains the package `sam3`)")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--checkpoint-status", type=Path, default=Path("/root/autodl-tmp/sam3_preparation/download_status.json"))
    p.add_argument("--shard", type=lambda s: tuple(int(v) for v in s.split("/")), default=(0, 1))
    p.add_argument("--limit", type=int)
    p.add_argument("--memory-fraction", type=float, default=0.3)
    p.add_argument("--foris", help="episodes.jsonl of a FoRIS run on the same manifest, for the paired difference")
    p.add_argument("--merge", action="store_true")
    p.add_argument("--score", action="store_true", help="CPU only, after whole-stage PREDICTIONS_FROZEN")
    p.add_argument("--preflight", action="store_true", help="CPU paired UID/header contract; no annotation pixels/model")
    p.add_argument("--reuse-predictions-from", type=Path, help="exact engineering smoke prefix, not scored records")
    p.add_argument("--resume", action="store_true", help="same-code verified completed-case prefix; preserve old errors")
    p.add_argument("--keep-candidates", action="store_true", help="CPU score smoke: retain prefix candidates until DEV reuse")
    p.add_argument("--candidate-budget-bytes", type=int, default=2*1024**3)
    p.add_argument("--fixture", action="store_true")
    p.add_argument("--unguarded", action="store_true")
    a = p.parse_args()
    if a.fixture:
        fixture()
    if sum((a.merge, a.score, a.preflight)) > 1:
        p.error("choose one of --merge/--score/--preflight")
    (preflight if a.preflight else merge if a.merge else score if a.score else work)(a)


if __name__ == "__main__":
    main()
