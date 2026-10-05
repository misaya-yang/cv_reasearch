#!/usr/bin/env python3
"""How much of the reference mask reaches SAM3: variants of the first pass that hand over more than one box (GPU).

  python scripts/sam3_reference_use.py --run RUN/dev --manifest SUITE/dev_episodes.json --sam3 SRC --checkpoint CKPT --out RUN/dev/reference_use

The stored first pass gives SAM3 one box around one connected component of the reference mask, drawn at random.
Variants (same canvas layout, same query, no query annotation opened; proposals stored like the first pass):
  maskout    the reference pixels outside the mask are set to grey; the same single box
  allboxes   one positive box per component of the reference mask (the 6 largest)
  both       maskout and allboxes
  negative   the single box, plus negative boxes on SAM3's own first-pass reference-side proposals that lie off the mask
  zoom       the reference rectangle shows only the exemplar's box with a 25% margin; zoom_maskout: also grey outside the mask
  refflip    the reference mirrored; queryflip: the query mirrored (proposals stored mirrored back)
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
VARIANTS = ("maskout", "allboxes", "both", "negative")
VIEWS = ("zoom", "zoom_maskout", "refflip", "queryflip")  # other views of the same pair, for layout and for voting across views


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True); p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)
    p.add_argument("--sam3", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--limit", type=int); p.add_argument("--shards", type=int, nargs="*")
    p.add_argument("--variants", nargs="*", default=list(VARIANTS))
    a = p.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from scipy import ndimage
    import sam3_stitch as S
    man = json.loads(Path(a.manifest).read_text())
    data, ann = Path(man["data_root"]), Path(man["annotation_root"])
    paths = sorted(Path(a.run).glob("predictions_shard*.jsonl"))
    if a.shards is not None:
        paths = [q for q in paths if int(q.stem.split("shard")[1]) in a.shards]
    recs = [r for q in paths for r in map(json.loads, q.read_text().splitlines()) if r.get("candidate_file")][:a.limit]
    out = Path(a.out)
    for v in a.variants:
        (out / v / "candidates").mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(a.sam3))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    model = build_sam3_image_model(bpe_path=str(Path(a.sam3) / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"), device="cuda",
                                   checkpoint_path=str(a.checkpoint), load_from_HF=False,
                                   eval_mode=True, enable_inst_interactivity=False, compile=False).float().eval()
    S.configure_fp32_mlp_backend(model)
    proc = Sam3Processor(model, device="cuda", confidence_threshold=.5)
    s, t = S.rectangles()

    def run(canvas, boxes):
        """boxes: [(normalised cxcywh, positive)]. The 20 most confident masks in the query rectangle, their scores, areas in both rectangles."""
        state = proc.set_image(canvas)
        proc.reset_all_prompts(state)
        seen, inner = {}, proc.model.forward_grounding

        def spy(**kw):
            seen["out"] = inner(**kw)
            return seen["out"]
        proc.model.forward_grounding = spy
        try:
            for box, label in boxes:
                state = proc.add_geometric_prompt(state=state, box=box, label=label)
        finally:
            proc.model.forward_grounding = inner
        o = seen["out"]
        prob = (o["pred_logits"].sigmoid() * o["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)[0]
        top = prob.argsort(descending=True, stable=True)[:S.KEEP]
        raw = torch.nn.functional.interpolate(o["pred_masks"][0, top][:, None].float(), (S.CANVAS, S.CANVAS), mode="bilinear", align_corners=False)[:, 0] > 0
        return prob[top].float().cpu().tolist(), raw

    def norm(box, size):  # [x, y, w, h] in reference pixels -> normalised cxcywh on the canvas
        sx, sy = s[2] / size[0], s[3] / size[1]
        x, y, w, h = box
        return [(x * sx + s[0] + w * sx / 2) / S.CANVAS, (y * sy + s[1] + h * sy / 2) / S.CANVAS, w * sx / S.CANVAS, h * sy / S.CANVAS]

    streams = {v: open(out / v / "variant.jsonl", "w") for v in a.variants}
    start = time.monotonic()
    with torch.inference_mode():
        for n, r in enumerate(recs):
            ref, query = (Image.open(data / r[k]).convert("RGB") for k in ("support", "query"))
            mask = np.asarray(Image.open(ann / Path(r["support"]).with_suffix(".png"))) == r["c"] + 1  # the reference label only
            lab, count = ndimage.label(mask, structure=np.ones((3, 3)))
            area = ndimage.sum(mask, lab, range(1, count + 1))
            comps = [ndimage.find_objects(lab)[i] for i in np.argsort(-area)[:6]]
            every = [norm([float(xs.start), float(ys.start), float(xs.stop - xs.start), float(ys.stop - ys.start)], ref.size) for ys, xs in comps]
            one = norm(r["exemplar_box"], ref.size)
            grey = np.asarray(ref).copy(); grey[~mask] = 127
            plain, masked = S.stitch(ref, query, r["exemplar_box"])[0], S.stitch(Image.fromarray(grey), query, r["exemplar_box"])[0]
            for v in a.variants:
                boxes = [(one, True)]
                if v in ("allboxes", "both"):
                    boxes = [(b, True) for b in every]
                if v == "negative":
                    ts = torch.from_numpy(S.on_canvas(mask, s)).cuda()
                    prob0, raw0 = run(plain, [(one, True)])
                    for j in range(len(prob0)):
                        m = raw0[j]
                        inref = m[s[1]:s[1] + s[3], s[0]:s[0] + s[2]]
                        if int(m[t[1]:t[1] + t[3], t[0]:t[0] + t[2]].sum()) == 0 and int(inref.sum()) > 0 and int((inref & ts).sum()) == 0 and len(boxes) < 4:
                            ys, xs = torch.nonzero(inref, as_tuple=True)
                            x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
                            boxes.append(([(s[0] + (x0 + x1) / 2) / S.CANVAS, (s[1] + (y0 + y1) / 2) / S.CANVAS, (x1 - x0) / S.CANVAS, (y1 - y0) / S.CANVAS], False))
                canvas = masked if v in ("maskout", "both") else plain
                if v.startswith("zoom"):
                    x, y, w, h = r["exemplar_box"]
                    x0, y0 = max(0, int(x - .25 * w)), max(0, int(y - .25 * h))
                    x1, y1 = min(ref.size[0], int(x + 1.25 * w) + 1), min(ref.size[1], int(y + 1.25 * h) + 1)
                    crop = Image.fromarray((grey if v == "zoom_maskout" else np.asarray(ref))[y0:y1, x0:x1])
                    canvas, box = S.stitch(crop, query, [x - x0, y - y0, w, h])
                    boxes = [(box, True)]
                if v == "refflip":
                    x, y, w, h = r["exemplar_box"]
                    canvas, box = S.stitch(ref.transpose(Image.FLIP_LEFT_RIGHT), query, [ref.size[0] - x - w, y, w, h])
                    boxes = [(box, True)]
                if v == "queryflip":
                    canvas = S.stitch(ref, query.transpose(Image.FLIP_LEFT_RIGHT), r["exemplar_box"])[0]
                prob, raw = run(canvas, boxes)
                q, ref_part = raw[:, t[1]:t[1] + t[3], t[0]:t[0] + t[2]], raw[:, s[1]:s[1] + s[3], s[0]:s[0] + s[2]]
                if v == "queryflip":
                    q = torch.flip(q, dims=[2])
                name = "candidates/%d_%d_%d.npz" % (r["fold"], r["e"], r["c"])
                np.savez_compressed(out / v / name, proposal_query=np.packbits(q.cpu().numpy().reshape(len(prob), t[2] * t[3]), axis=1))
                streams[v].write(json.dumps(dict(key=[r["fold"], r["e"], r["c"]], boxes=len(boxes), candidate_file=name, proposal_shape=[len(prob), t[3], t[2]],
                                                 meta=[[float(prob[i]), int(q[i].sum()), int(ref_part[i].sum())] for i in range(len(prob))])) + "\n")
                streams[v].flush()
            if n % 50 == 0:
                print("%d/%d episodes, %.2f s per episode" % (n + 1, len(recs), (time.monotonic() - start) / (n + 1)), flush=True)
    print(json.dumps(dict(state="COMPLETED", episodes=len(recs), variants=a.variants, elapsed_s=round(time.monotonic() - start, 1), query_annotation_opened=False)))


if __name__ == "__main__":
    main()
