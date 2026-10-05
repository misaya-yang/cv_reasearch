#!/usr/bin/env python3
"""Prompt transfer without a canvas: the exemplar is encoded on the reference image, the query is decoded with it (GPU).

  python scripts/sam3_transfer.py --run RUN/dev --manifest SUITE/dev_episodes.json --sam3 SRC --checkpoint CKPT --out RUN/dev/reference_use

The stitched canvas gives each image a strip of SAM3's input. Here both images pass the backbone at full input size;
SAM3's geometry encoder turns the reference box(es) into prompt tokens on the reference features, and the encoder,
decoder and mask head run on the query features with those tokens. Variants: transfer (the stored single box),
transfer_all (one box per component of the reference mask, the 6 largest). Proposals are stored like the first pass
(at the canvas's query-rectangle size). No query annotation is opened.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
VARIANTS = ("transfer", "transfer_all")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True); p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)
    p.add_argument("--sam3", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from scipy import ndimage
    import sam3_stitch as S
    man = json.loads(Path(a.manifest).read_text())
    data, ann = Path(man["data_root"]), Path(man["annotation_root"])
    recs = [r for q in sorted(Path(a.run).glob("predictions_shard*.jsonl")) for r in map(json.loads, q.read_text().splitlines()) if r.get("candidate_file")][:a.limit]
    out = Path(a.out)
    for v in VARIANTS:
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
    t = S.rectangles()[1]
    text = model.backbone.forward_text(["visual"], device="cuda")

    def features(image):
        b = proc.set_image(image)["backbone_out"]
        b.update(text)
        return b

    def decode(ref_b, query_b, boxes):
        geo = model._get_dummy_prompt()
        for box in boxes:
            geo.append_boxes(torch.tensor(box, device="cuda", dtype=torch.float32).view(1, 1, 4), torch.tensor([True], device="cuda").view(1, 1))
        prompt, mask, _ = model._encode_prompt(ref_b, proc.find_stage, geo)
        query_b, enc, _ = model._run_encoder(query_b, proc.find_stage, prompt, mask)
        o = {"encoder_hidden_states": enc["encoder_hidden_states"], "prev_encoder_out": {"encoder_out": enc, "backbone_out": query_b}}
        o, hs = model._run_decoder(memory=o["encoder_hidden_states"], pos_embed=enc["pos_embed"], src_mask=enc["padding_mask"], out=o,
                                   prompt=prompt, prompt_mask=mask, encoder_out=enc)
        ids = proc.find_stage.img_ids
        if "id_mapping" in query_b and query_b["id_mapping"] is not None:
            ids = query_b["id_mapping"][ids]
        model._run_segmentation_heads(out=o, backbone_out=query_b, img_ids=ids, vis_feat_sizes=enc["vis_feat_sizes"],
                                      encoder_hidden_states=o["encoder_hidden_states"], prompt=prompt, prompt_mask=mask, hs=hs)
        prob = (o["pred_logits"].sigmoid() * o["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)[0]
        top = prob.argsort(descending=True, stable=True)[:S.KEEP]
        raw = torch.nn.functional.interpolate(o["pred_masks"][0, top][:, None].float(), (t[3], t[2]), mode="bilinear", align_corners=False)[:, 0] > 0
        return prob[top].float().cpu().tolist(), raw

    streams = {v: open(out / v / "variant.jsonl", "w") for v in VARIANTS}
    start = time.monotonic()
    with torch.inference_mode():
        for n, r in enumerate(recs):
            ref, query = (Image.open(data / r[k]).convert("RGB") for k in ("support", "query"))
            mask = np.asarray(Image.open(ann / Path(r["support"]).with_suffix(".png"))) == r["c"] + 1  # the reference label only
            W, H = ref.size
            norm = lambda b: [(b[0] + b[2] / 2) / W, (b[1] + b[3] / 2) / H, b[2] / W, b[3] / H]
            lab, count = ndimage.label(mask, structure=np.ones((3, 3)))
            area = ndimage.sum(mask, lab, range(1, count + 1))
            every = [norm([float(xs.start), float(ys.start), float(xs.stop - xs.start), float(ys.stop - ys.start)])
                     for ys, xs in (ndimage.find_objects(lab)[i] for i in np.argsort(-area)[:6])]
            ref_b, query_b = features(ref), features(query)
            for v, boxes in (("transfer", [norm(r["exemplar_box"])]), ("transfer_all", every)):
                prob, q = decode(ref_b, query_b, boxes)
                name = "candidates/%d_%d_%d.npz" % (r["fold"], r["e"], r["c"])
                np.savez_compressed(out / v / name, proposal_query=np.packbits(q.cpu().numpy().reshape(len(prob), t[2] * t[3]), axis=1))
                streams[v].write(json.dumps(dict(key=[r["fold"], r["e"], r["c"]], boxes=len(boxes), candidate_file=name, proposal_shape=[len(prob), t[3], t[2]],
                                                 meta=[[float(prob[i]), int(q[i].sum()), 0] for i in range(len(prob))])) + "\n")
                streams[v].flush()
            if n % 50 == 0:
                print("%d/%d episodes, %.2f s per episode" % (n + 1, len(recs), (time.monotonic() - start) / (n + 1)), flush=True)
    print(json.dumps(dict(state="COMPLETED", episodes=len(recs), variants=VARIANTS, elapsed_s=round(time.monotonic() - start, 1), query_annotation_opened=False)))


if __name__ == "__main__":
    main()
