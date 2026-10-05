#!/usr/bin/env python3
"""Re-read the shortlisted names on the reference object's own crop (GPU): SAM3 as the judge of an object-centred image.

  python scripts/sam3_crop_name.py --run RUN/dev --manifest SUITE/dev_episodes.json --variant sem_name --sam3 SRC --checkpoint CKPT --out X/crop_dev.jsonl

The shortlist of names was scored on the features of the whole reference image. Here the reference object is cropped
(box of its mask, 20% margin) and shown to SAM3 at full input size, once as it is and once with everything outside
the mask set to grey; each shortlisted name gets its top confidence and the IoU of its best instance, of its semantic
map and of the union of its confident instances with the mask. No query label is used.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True); p.add_argument("--manifest", required=True); p.add_argument("--out", required=True)
    p.add_argument("--variant", default="sem_name"); p.add_argument("--sam3", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--limit", type=int)
    a = p.parse_args()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torchvision.transforms import v2
    import sam3_stitch as S
    man = json.loads(Path(a.manifest).read_text())
    data, ann = Path(man["data_root"]), Path(man["annotation_root"])
    rows = {tuple(j["key"]): j for j in map(json.loads, (Path(a.run) / "reference_use" / a.variant / "variant.jsonl").read_text().splitlines())}
    recs = [r for q in sorted(Path(a.run).glob("predictions_shard*.jsonl")) for r in map(json.loads, q.read_text().splitlines())
            if (r["fold"], r["e"], r["c"]) in rows][:a.limit]
    sys.path.insert(0, str(a.sam3))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.data_misc import FindStage
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)
    model = build_sam3_image_model(bpe_path=str(Path(a.sam3) / "sam3/assets/bpe_simple_vocab_16e6.txt.gz"), device="cuda",
                                   checkpoint_path=str(a.checkpoint), load_from_HF=False,
                                   eval_mode=True, enable_inst_interactivity=False, compile=False).float().eval()
    S.configure_fp32_mlp_backend(model)
    proc = Sam3Processor(model, device="cuda", confidence_threshold=.5)

    def ground(image, names):
        n = len(names)
        b = model.backbone.forward_image(proc.transform(v2.functional.to_image(image).to("cuda")).unsqueeze(0))
        text = {k: v for k, v in model.backbone.forward_text(names, device="cuda").items() if torch.is_tensor(v)}
        stage = FindStage(img_ids=torch.zeros(n, device="cuda", dtype=torch.long), text_ids=torch.arange(n, device="cuda"),
                          input_boxes=None, input_boxes_mask=None, input_boxes_label=None, input_points=None, input_points_mask=None)
        bo = {**b, **text}
        prompt, mask, bo = model._encode_prompt(bo, stage, model._get_dummy_prompt(n))
        bo, enc, _ = model._run_encoder(bo, stage, prompt, mask)
        o = {"encoder_hidden_states": enc["encoder_hidden_states"], "prev_encoder_out": {"encoder_out": enc, "backbone_out": bo}}
        o, hs = model._run_decoder(memory=o["encoder_hidden_states"], pos_embed=enc["pos_embed"], src_mask=enc["padding_mask"], out=o,
                                   prompt=prompt, prompt_mask=mask, encoder_out=enc)
        presence = o["presence_logit_dec"].sigmoid().flatten()[:n] if o["presence_logit_dec"].numel() >= n else o["presence_logit_dec"].sigmoid().flatten().expand(n)
        prob = (o["pred_logits"].sigmoid() * o["presence_logit_dec"].sigmoid().unsqueeze(1)).squeeze(-1)
        ids = stage.img_ids
        if "id_mapping" in bo and bo["id_mapping"] is not None:
            ids = bo["id_mapping"][ids]
        model._run_segmentation_heads(out=o, backbone_out=bo, img_ids=ids, vis_feat_sizes=enc["vis_feat_sizes"],
                                      encoder_hidden_states=o["encoder_hidden_states"], prompt=prompt, prompt_mask=mask, hs=hs)
        return prob.float(), o["pred_masks"].float() > 0, o["semantic_seg"].float(), presence.float()
    start = time.monotonic()
    with open(a.out, "w") as stream, torch.inference_mode():
        for n, r in enumerate(recs):
            j = rows[(r["fold"], r["e"], r["c"])]
            names = [v[0] for v in j["rerank"]]
            ref = Image.open(data / r["support"]).convert("RGB")
            mask = np.asarray(Image.open(ann / Path(r["support"]).with_suffix(".png"))) == r["c"] + 1  # the reference label only
            ys, xs = np.nonzero(mask)
            bw, bh = xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min()
            x0, y0 = max(0, int(xs.min() - .2 * bw)), max(0, int(ys.min() - .2 * bh))
            x1, y1 = min(mask.shape[1], int(xs.max() + 1 + .2 * bw)), min(mask.shape[0], int(ys.max() + 1 + .2 * bh))
            crop, cm = ref.crop((x0, y0, x1, y1)), mask[y0:y1, x0:x1]
            grey = np.asarray(crop).copy(); grey[~cm] = 124
            out = {}
            for tag, image in (("crop", crop), ("grey", Image.fromarray(grey))):
                prob, m, sem, presence = ground(image, names)
                target = F.interpolate(torch.from_numpy(cm).float().cuda()[None, None], m.shape[-2:], mode="area")[0, 0] > .5
                rel = lambda x: (x & target).flatten(-2).sum(-1).float() / ((x | target).flatten(-2).sum(-1).float()).clamp(min=1)
                iou = rel(m)                                      # [names, queries]
                k = (prob * iou).argmax(1, keepdim=True)
                tc = prob.max(1).values
                sm = F.interpolate(sem.reshape(len(names), 1, *sem.shape[-2:]), m.shape[-2:], mode="bilinear", align_corners=False)[:, 0] > 0
                um = (m & (prob >= .7 * tc[:, None])[:, :, None, None]).any(1)
                out[tag] = [[round(float(tc[i]), 4), round(float(prob[i, k[i, 0]]), 4), round(float(iou[i, k[i, 0]]), 4), round(float(rel(sm)[i]), 4),
                             round(float(rel(um)[i]), 4), round(float(presence[i]), 4)] for i in range(len(names))]
            stream.write(json.dumps(dict(key=[r["fold"], r["e"], r["c"]], names=names, **out)) + "\n"); stream.flush()
            if n % 40 == 0:
                print("%d/%d, %.2f s per episode" % (n + 1, len(recs), (time.monotonic() - start) / (n + 1)), flush=True)
    print(json.dumps(dict(state="COMPLETED", episodes=len(recs), elapsed_s=round(time.monotonic() - start, 1))))


if __name__ == "__main__":
    main()
